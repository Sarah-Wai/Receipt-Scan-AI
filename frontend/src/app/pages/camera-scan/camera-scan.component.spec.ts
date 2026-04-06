import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CameraScanComponent } from './camera-scan.component';

describe('CameraScanComponent', () => {
  let component: CameraScanComponent;
  let fixture: ComponentFixture<CameraScanComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CameraScanComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CameraScanComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
