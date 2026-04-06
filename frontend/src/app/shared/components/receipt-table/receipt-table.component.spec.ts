import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ReceiptTableComponent } from './receipt-table.component';

describe('ReceiptTableComponent', () => {
  let component: ReceiptTableComponent;
  let fixture: ComponentFixture<ReceiptTableComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ReceiptTableComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ReceiptTableComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
